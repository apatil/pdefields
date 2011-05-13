      subroutine gmrfmetro (ind,dat,ptr,nnz,nx,x,lp,
     & diag, acc, norms, M, lv, nv)
!f2py intent(hide) nnz, nx, nv
!f2py intent(inplace) lp, x
      implicit none
      double precision x(nx), dat(nnz), lp(nx)
      double precision diag(nx), acc(nx), norms(nx), mc(nx), M(nx)
      double precision xp, lpp, lv(nx,nv)
      integer ind(nnz), ptr(nx+1), nx, nnz, i, j, nv
      do i=1,nx
          mc(i) = diag(i)*x(i)
          do j=ptr(i)+1,ptr(i+1)
              mc(i) = mc(i) - dat(j)*x(ind(j)+1)
          end do
          mc(i) = mc(i)/diag(i)
          
          xp = norms(i)+mc(i)
          {LPF}
          
          if ((lpp-lp(i)).GT.dlog(acc(i))) then
              lp(i)=lpp
              x(i)=xp
          end if
          
      end do
      
      return
      end
  